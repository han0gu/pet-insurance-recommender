from langchain_core.documents import Document

chunk = Document(
    page_content=('【펫샵】 동물보호법 시행규칙에 따라 동물을 분양하는 영업활동을 할 수 있는 영업자를 말합니다. 【분양】 펫샵에 유상의 재화를 제공하고 '
 '반려동물을 입양하는 행위를 말합니다.\n'
 '② 제1항에도 불구하고, 암, 백내장, 녹내장, 심장질환, 신장질환, 방광질환 및 각종결석이 대기기간 중에 발생한 손해에 대해서는 '
 '보상하지 않습니다.\n'
 '제2조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관 및 해당 특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000228',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
