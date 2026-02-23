from langchain_core.documents import Document

chunk = Document(
    page_content=('1)신경계# 가) “신경계에 장해를 남긴 때”라 함은 뇌, 척수 및 말초신경계 손상으법\n'
 '로 “<붙임>일상생활 기본동작(ADLs) 제한 장해평가표”의 5가지 기 ㆍ\n'
 '본동작중 하나 이상의 동작이 제한되었을 때를 말한다. 규정\n'
 '나) 위 가)의 경우 “<붙임>일상생활 기본동작(ADLs) 제한 장해평가표”\n'
 '상 지급률이 10% 미만인 경우에는 보장대상이 되는 장해로 인정하지\n'
 '않는다.\n'
 '다) 신경계의 장해로 발생하는 다른 신체부위의 장해(눈, 귀, 코, 팔,\n'
 '다리 등)는 해당 장해로도 평가하고 그 중 높은 지급률을 적용한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000932',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
