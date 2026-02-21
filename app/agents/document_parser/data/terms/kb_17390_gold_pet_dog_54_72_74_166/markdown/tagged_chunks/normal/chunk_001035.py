from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '|  | 803 순환기계질환 | 질병 세부 질병명 |\n'
 '| 심장사상충 감염 | 803 순환기계질환 |  |\n'
 '| 대동맥판폐쇄부전 | 803 순환기계질환 |  |\n'
 '| 대동맥협착증 | 803 순환기계질환 |  |\n'
 '| 동맥관개존증 | 803 순환기계질환 |  |\n'
 '| 전도 장애 | 803 순환기계질환 |  |\n'
 '| 삼첨판폐쇄부전 | 803 순환기계질환 |  |\n'
 '| 심근증 | 803 순환기계질환 |  |\n'
 '| 심낭수종 | 803 순환기계질환 |  |\n'
 '| 심내막염 | 803 순환기계질환 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001035',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
