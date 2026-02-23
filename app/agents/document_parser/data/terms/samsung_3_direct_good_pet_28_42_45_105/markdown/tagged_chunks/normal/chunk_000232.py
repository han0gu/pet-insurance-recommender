from langchain_core.documents import Document

chunk = Document(
    page_content=('법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.| · 추심명령 : | 채무자가 제3채무자에 대하여 '
 '가지고 있는 금전채권을 대위의 절차 없이 채무자에 갈음하여 직접 추심(받아냄)할 수 있는 권리를 부여하는 집행법원의 결정 |\n'
 '| --- | --- |\n'
 '| · 전부명령 : | 채무자가 제3채무자에 대한 채권을 채권자에게 이전시키고 그 대신 채무자에 대한 채권이 소멸되는 집행법원의 결정 |'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000232',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
