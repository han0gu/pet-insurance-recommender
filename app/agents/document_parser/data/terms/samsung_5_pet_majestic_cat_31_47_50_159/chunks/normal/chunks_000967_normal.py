from langchain_core.documents import Document

chunk = Document(
    page_content=('5) "흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때" 라 함은 아래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '가) 방광의 용량이 50cc 이하로 위축되었거나 요도협착, 배뇨기능 상실로 영구 적인 간헐적 인공요도가 필요한 때 나) 음경의 1/2 '
 '이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때 다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호흡곤란으로 지속적인 산소 치료가 '
 '필요하며, 폐기능 검사(PFT)상 폐환기 기능(1초간 노력성 호기량, FEV1)이 정상예측치의 40% 이하로 저하된 때'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['urinary', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000967',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
