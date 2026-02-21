from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3 | 순환기 질환 | HAA013 | 폐동맥 협착 · PS |\n'
 '| 3 | 순환기 질환 | HAA019 | 기타 선천성 심장 질환 (확진된) 심장사상충증 |\n'
 '| 3 | 순환기 질환 | HAA020 | 기타 심혈관계 질환 |\n'
 '| 3 | 순환기 질환 | HAA021 HAA022 | 점액성 이첨판막변성 |\n'
 '| 3 | 순환기 질환 | HAA023 | 고양이 비대성 심근병증 |\n'
 '| 4 | 비뇨기과 | AGA001 | 신장의 양성 신생물 |\n'
 '| 4 | 비뇨기과 | AGB001 | 신장의 악성 신생물 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
