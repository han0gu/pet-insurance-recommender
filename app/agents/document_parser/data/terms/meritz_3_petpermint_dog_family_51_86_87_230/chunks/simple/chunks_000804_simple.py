from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 방광의 용량이 50cc 이하로 위축되었거나 요도협 착, 배뇨기능 상실로 영구적인 간헐적 인공요도 가 필요한 때 나) 음경의 1/2 '
 '이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때 다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호 흡곤란으로 지속적인 산소치료가 '
 '필요하며, 폐기 능 검사(PFT)상 폐환기 기능(1초간 노력성 호기 량, FEV1)이 정상예측치의 40% 이하로 저하된 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 225},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['urinary',
                             'dental',
                             'skin',
                             'digestive',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000804',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
