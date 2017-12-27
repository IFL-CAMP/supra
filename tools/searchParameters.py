import sys
import os
import fnmatch
import re
import pyparsing as pp
import shutil

filename_patterns = ["*.cu", "*.c", "*.cpp", "*.cuh", "*.h"]
parameter_pattern = "m_valueRangeDictionary.set"
parameter_detail_pattern = "m_valueRangeDictionary.set<(?P<type>[^>]+)>(?P<content>\\([^;]+\\));"
output_dir = "../doc/supraParameters"
output_filename = "../doc/supraParameters/params{0}.h"


def filter_function(e, predicate):
    if isinstance(e, list):
        return list(filter(None, map(lambda x: filter_function(x, predicate), e)))
    elif predicate(e):
        return e


def get_parameter_allowed_value_string(par):
    if par['type'] == 'unrestricted':
        return 'any'
    elif par['type'] == 'closed':
        return 'any in range [' + par['lower_bound'] + ', ' + par['upper_bound'] + ']'
    elif par['type'] == 'discrete':
        return 'any in set {' + ', '.join(par['range']) + '}'
    else:
        pass

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

start_dir = os.path.abspath(os.path.join(os.curdir, "../src"))
code_files = []
for paths, dirs, filenames in os.walk(os.path.abspath(start_dir)):
    for filename_pattern in filename_patterns:
        for filename_filtered in fnmatch.filter(filenames, filename_pattern):
            filename_complete = os.path.join(paths, filename_filtered)
            code_files.append(filename_complete)

pattern = re.compile(parameter_pattern)
parameter_strings_collection = {}
for filename in code_files:
    lines_parameters = [line.rstrip('\n') for line in open(filename, encoding='utf8') if pattern.search(line) is not None]
    if len(lines_parameters) > 0:
        filebase = os.path.basename(filename)
        parameter_strings_collection[filebase] = lines_parameters

thecontent = pp.Word(pp.alphanums + '+' + '-' + '*' + '/' + '_' + '.' + ' ' + '"' + "'" + '[' + ']') | ','
parens = pp.nestedExpr( pp.Literal('(') ^ pp.Literal('{'), pp.Literal(')') ^ pp.Literal('}'), content=thecontent)
num_params = 0
parameters = {}
pattern_details = re.compile(parameter_detail_pattern)
for basename in parameter_strings_collection.keys():
    classname = basename.split('.')
    classname = classname[0]

    parameter_strings = parameter_strings_collection[basename]
    for parameter_string in parameter_strings:
        parameter_details = pattern_details.search(parameter_string)
        if parameter_details is not None:
            parameter_type = parameter_details.group("type")
            parameter_content = parameter_details.group("content")

            #print(parameter_content)
            parsed = parens.parseString(parameter_content).asList()[0]
            parsed = list(filter_function(parsed, lambda x: not (x == ',')))
            #print("" + str(len(parsed)) + " " + str(parsed))

            if len(parsed) == 3:
                parameter = { "type": "unrestricted", "data_type": parameter_type,
                              "name": parsed[0].strip('"'), "display_name": parsed[2].strip('"'),
                              "default_value": parsed[1]}
            elif len(parsed) == 5:
                parameter = {"type": "closed", "data_type": parameter_type,
                             "name": parsed[0].strip('"'), "display_name": parsed[4].strip('"'),
                             "default_value": parsed[3], "lower_bound": parsed[1], "upper_bound": parsed[2]}
            elif len(parsed) == 4:
                parameter = {"type": "discrete", "data_type": parameter_type,
                             "name": parsed[0].strip('"'), "display_name": parsed[3].strip('"'),
                             "default_value": parsed[2], "range": parsed[1]}

            if classname not in parameters.keys():
                parameters[classname] = {}
            parameters[classname][parameter['name']] = parameter
            num_params += 1


for classname in sorted(parameters.keys()):
    lines = []
    classlines = [
        '/** \\nosubgrouping',
        '  */',
        'class ' + classname,
        '{',
        'public:',
        '    /** @name Parameters',# + classname,
        '     *  Parameters of node type ' + classname + '.',
        '     */',
        '    /**  @{',
        '     */'
    ]
    for parameter_name in sorted(parameters[classname].keys()):
        p = parameters[classname][parameter_name]
        classlines = classlines + \
                     ['    /// \\brief Parameter {0}: {1}'.format(p['name'], p['display_name']),
                      '    ///',
                      '    /// Default value: ' + p['default_value'] + '<BR>',
                      '    /// Valid values: ' + get_parameter_allowed_value_string(p),
                      '    ' + p['data_type'] + ' ' + p['name'] + ' = ' + p['default_value'] + ';', '']

    classlines = classlines + [
        '    /** @} */',
        '}', ''
    ]
    lines = lines + classlines
    lines = list(map(lambda l: '    ' + l, lines))
    lines = ['namespace supra', '{'] + lines + ['}']
    lines = list(map(lambda l: l + '\n', lines))
    out_header = open(output_filename.format(classname), 'w')
    out_header.writelines(lines)
    out_header.close()


#print(num_params)
#print(parameters)