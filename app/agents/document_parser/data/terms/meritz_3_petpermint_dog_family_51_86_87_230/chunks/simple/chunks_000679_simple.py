from langchain_core.documents import Document

chunk = Document(
    page_content=('HAA007 | 확장성 심근병증\n'
 'HAA008 | 비대성 심근병증\n'
 'HAA009 | 제한성 심근병증 일시적 심근비대증\n'
 'HAA010 HAA011 | 기타 심근증\n'
 'HAA012 | 대동맥 협착증 · AS\n'
 'HAA013 | 폐동맥 협착 · PS\n'
 'HAA019 | 기타 선천성 심장 질환 (확진된)\n'
 'HAA020 | 심장사상충증\n'
 'HAA021 | 기타 심혈관계 질환\n'
 'HAA022 | 점액성 이첨판막변성\n'
 '4 | 비뇨기과 질환 | AGA001 | 신장의 양성 신생물 신장의 악성 신생물\n'
 'AGB001'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 196},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000679',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
