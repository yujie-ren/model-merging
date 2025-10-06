import re




def main():
    print("Explore regular expression")

    name_LLM = "model.layers.21.post_attention_layernorm.weight"
    match = re.search(r'layers\.(\d+)\.', name_LLM)
    print(match)
    print(type(match))

    print(match.group())
    print(type(match.group()))

    print(match.group(1))
    layer_number = int(match.group(1))
    print(layer_number)

    print("done")












if __name__ == "__main__":
    main()