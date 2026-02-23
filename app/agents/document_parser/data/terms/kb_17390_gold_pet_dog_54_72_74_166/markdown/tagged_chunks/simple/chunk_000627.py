from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우 이 특별약관의 보험가입금액을 무지개다리위로금(강아지, 사망)으로 보험수\n'
 '- 익자에게 지급합니다.\n'
 '- \uf000 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경\n'
 '- 우 동물병원에서 발급한 소견서를 제출하여야 합니다.\n'
 '- \uf000 제1항의 경우 무지개다리위로금보장개시일은 계약일로부터 그날을 포함하여 30일\n'
 '| 이 지난날의 | 다음날로 합니다. 단, 계약일은 제1회 보험료를 받은 날로 합니다. |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000627',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
