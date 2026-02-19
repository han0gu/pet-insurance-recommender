from langchain_core.documents import Document

chunk = Document(
    page_content=('가) "신경계에 장해를 남긴 때" 라 함은 뇌, 척수 및 말초신경계 손상으로 "< 붙임> 일상생활 기본동작(ADLs) 제한 장해평가표" '
 '의 5가지 기본동작중 하나 이상의 동작이 제한되었을 때를 말한다. 나) 위 가)의 경우 "<붙임> 일상생활 기본동작(ADLs) 제한 '
 '장해평가표" 상 지 급률이 10% 미만인 경우에는 보장대상이 되는 장해로 인정하지 않는다. 다) 신경계의 장해로 발생하는 다른 신체부위의 '
 '장해(눈, 귀, 코, 팔, 다리 등)는 해당 장해로도 평가하고 그 중 높은 지급률을 적용한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000971',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
