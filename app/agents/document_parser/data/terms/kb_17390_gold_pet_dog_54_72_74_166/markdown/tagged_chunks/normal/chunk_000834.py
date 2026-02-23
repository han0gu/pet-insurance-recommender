from langchain_core.documents import Document

chunk = Document(
    page_content=('기 ⑬ 신경계․정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다.\n'
 '다만, 좌․우의 눈, 귀, 팔, 다리, 손가락, 발가락은 각각 다른 신체부위로 본다.# 3. 기타1) 하나의 장해가 관찰 방법에 따라서 '
 '장해분류표상 2가지 이상의 신체부위에서- 장해로 평가되는 경우에는 그 중 높은 지급률을 적용한다.\n'
 '- 2) 동일한 신체부위에 2가지 이상의 장해가 발생한 경우에는 합산하지 않고 그\n'
 '- 중 높은 지급률을 적용함을 원칙으로 한다. 그러나 각 신체부위별 판정기준'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000834',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
