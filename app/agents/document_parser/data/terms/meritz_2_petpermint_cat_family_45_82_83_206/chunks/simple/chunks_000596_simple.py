from langchain_core.documents import Document

chunk = Document(
    page_content=('HAA011 | 기타 심근증\n'
 'HAA012 | 대동맥 협착증 · AS\n'
 'HAA013 | 폐동맥 협착 · PS\n'
 'HAA019 | 기타 선천성 심장 질환 (확진된) 심장사상충증\n'
 'HAA020 | 기타 심혈관계 질환\n'
 'HAA021 HAA022 | 점액성 이첨판막변성\n'
 'HAA023 | 고양이 비대성 심근병증\n'
 '4 | 비뇨기과 | AGA001 | 신장의 양성 신생물\n'
 'AGB001 | 신장의 악성 신생물\n'
 'AGC001 | 신장의 신생물 (양성 또는 악성이 불확실한)\n'
 'AGB002 | 이행상피세포암종 (방광)\n'
 'AGA003 | 기타 방광의 양성 신생물 기타'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 170},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000596',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
