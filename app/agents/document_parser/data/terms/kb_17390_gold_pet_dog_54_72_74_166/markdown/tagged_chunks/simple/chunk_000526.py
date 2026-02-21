from langchain_core.documents import Document

chunk = Document(
    page_content=('하여 해당 담보권을 실행하는 것을 말합니다. 해∙국세 및 지방세 체납처분 절차국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 '
 '경우 국세 기본법\n'
 '병\n'
 '및 지방세법에 의하여 체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산\n'
 '압류 등의 집행을 하는 것을 말합니다. 법원은 채권자의 신청에 따른 강제집행\n'
 '및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추심명령\n'
 '또는 전부명령에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다. 또 반'),
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
 'indexing': {'chunk_id': 'chunk_000526',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
