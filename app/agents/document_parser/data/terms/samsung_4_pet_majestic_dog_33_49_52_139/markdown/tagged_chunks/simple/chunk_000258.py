from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑤ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자가 통\n'
 '- 지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.\n'
 '# 제6관 계약의 해지 및 해약환급금 등# 제 32조 (계약자의 임의해지 및 피보험자의 서면동의 철회권)- ① 계약자는 계약이 소멸하기 '
 '전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는\n'
 '- 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '- ② 제22조(특별약관의 무효) 에 따라 사망을 보험금 지급사유로 하는 계약에서 서면으로'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
