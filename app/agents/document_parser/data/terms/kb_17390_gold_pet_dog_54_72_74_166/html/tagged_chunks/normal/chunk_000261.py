from langchain_core.documents import Document

chunk = Document(
    page_content=('. 법원은 채권자의 신청에 따른 강제집행 및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추심명령 또는 전부명령에 따라 '
 '회사는 채권자에게 해약환급금을 지급하게 됩니다'),
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
 'indexing': {'chunk_id': 'chunk_000261',
              'chunk_char_len': 102,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
