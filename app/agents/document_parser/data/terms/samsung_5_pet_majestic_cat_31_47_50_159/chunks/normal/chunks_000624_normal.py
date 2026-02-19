from langchain_core.documents import Document

chunk = Document(
    page_content=('추심명령 : 채무자가 제3채무자에 대하여 가지고 있는 금전채권을 대위의 절차 없이 채무자 에 갈음하여 직접 추심(받아냄)할 수 있는 '
 '권리를 부여하는 집행법원의 결정 전부명령 : 채무자가 제3채무자에 대한 채권을 채권자에게 이전시키고 그 대신 채무자에 대 한 채권이 '
 '소멸되는 집행법원의 결정\n'
 '[국세 및 지방세 체납처분 절차] 국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000624',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
