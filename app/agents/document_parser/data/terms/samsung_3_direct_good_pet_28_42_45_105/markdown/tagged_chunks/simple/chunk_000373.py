from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 갈음하여 직접 추심(받아냄)할 수 있는 권리를 부여하는 집행법원의 결정\n'
 '- · 전부명령 : 채무자가 제3채무자에 대한 채권을 채권자에게 이전시키고 그 대신 채무자에 대\n'
 '- 한 채권이 소멸되는 집행법원의 결정\n'
 '# [국세 및 지방세 체납처분 절차]국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하\n'
 '여 체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니\n'
 '다.'),
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
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
