import re
import toml
import sys
import ast

def parse_configuration_native(fname):
    '''
        Parse the native Genesis1.3v4 configuration file
    '''
    parsed_configuration = dict()
    current_section_name = None

    for (i, line) in enumerate(open(fname, "r")):
        stripped_line = line.strip().replace(" ", "")
        if current_section_name == None:
            if stripped_line == "":
                continue
            
            if stripped_line == "&end":
                print(fname+":","Unexpected field termination &end at line", i+1)
                exit(-1)
            
            if stripped_line[0]!="&":
                print(fname+":", "Unexpected orphan parameter:\"",
                      stripped_line[0],"\" at line", i+1)
            
            current_section_name = stripped_line[1:]
            parsed_configuration[current_section_name] = dict()
        else:
            if stripped_line == "":
                continue
            
            if stripped_line[0] == "&" and stripped_line != "&end":
                print(fname+":","Unexpected start of field", stripped_line,
                      "in already opened field", "&"+current_section_name,
                      "at line", i+1)
                exit(-1)

            if stripped_line == "&end":
                current_section_name = None
                continue
            
            splitted_line = stripped_line.split("=")
            parsed_configuration[current_section_name][splitted_line[0]] = splitted_line[1]
            

    assert current_section_name == None, "Not closed field &" + current_section_name
    return parsed_configuration

def parse_configuration_toml(fname):
    return toml.load(fname)

def parse_configuration_json(fname):
    return json.load(fname)

if __name__ == "__main__":
    print(parse_configuration_native(sys.argv[1]))
