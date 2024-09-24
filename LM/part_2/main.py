def main():
    model = LM_RNN(
            "lstm",
            env.args.emb_size, 
            env.args.hid_size, 
            len(env.lang)
        ).to(env.device)

if __name__ == "__main__":
    main()