from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3 | 순환기 질환 | HAA012 | 대동맥 협착증 · AS |\n'
 '| 3 | 순환기 질환 | HAA013 | 폐동맥 협착 · PS |\n'
 '| 3 | 순환기 질환 | HAA019 | 기타 선천성 심장 질환 (확진된) |\n'
 '| 3 | 순환기 질환 | HAA020 | 심장사상충증 |\n'
 '| 3 | 순환기 질환 | HAA021 | 기타 심혈관계 질환 |\n'
 '| 3 | 순환기 질환 | HAA022 | 점액성 이첨판막변성 |\n'
 '| 4 | 비뇨기과 질환 | AGA001 | 신장의 양성 신생물 신장의 악성 신생물 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000566',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
