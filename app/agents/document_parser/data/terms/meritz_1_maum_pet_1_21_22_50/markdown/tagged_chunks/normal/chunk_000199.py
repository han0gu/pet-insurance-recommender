from langchain_core.documents import Document

chunk = Document(
    page_content=('- 령할 수 있습니다.\n'
 '- ② 회사가 보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더라도\n'
 '- 회사는 이를 지급하지 않습니다.\n'
 '# 제6조(보험금의 청구)지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구하여\n'
 '야 합니다.- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증)\n'
 '- 4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서) 및 주민등록등본'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
