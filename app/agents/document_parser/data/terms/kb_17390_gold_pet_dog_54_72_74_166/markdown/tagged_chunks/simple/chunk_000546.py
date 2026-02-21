from langchain_core.documents import Document

chunk = Document(
    page_content=('| 기본형Ⅰ | 통원 | 1일당 15만원 한도 1일당 | 200만원 한도 | 및 1,000만원 질 |\n'
 '| 기본형Ⅱ | 입원 | 1일당 15만원 한도 1일당 | 250만원 한도 | 2,000만원 병 |\n'
 '| 기본형Ⅱ | 통원 | 1일당 15만원 한도 1일당 | 250만원 한도 | 2,000만원 |\n'
 '| 고급형Ⅰ | 입원 | 1일당 30만원 한도 1일당 | 250만원 한도 | 2,000만원 |\n'
 '| 고급형Ⅰ | 통원 | 1일당 30만원 한도 1일당 | 250만원 한도 | 2,000만원 반 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000546',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
