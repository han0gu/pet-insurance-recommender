from langchain_core.documents import Document

chunk = Document(
    page_content=('사. 수의사 : 「수의사법」 제2조(정의)에 따라 수의업무를 담당하는 사람으로서 농림축 산식품부장관의 면허를 받은 사람을 말합니다. 아. '
 '동물병원 : 「수의사법」 제2조(정의)에 따라 동물진료업을 하는 장소로서 「수의사 법」 제17조에 따른 신고를 한 국내 진료기관을 '
 '말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 1},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
