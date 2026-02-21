from langchain_core.documents import Document

chunk = Document(
    page_content=('(교합), 배열상태 및 아래턱의 개구(입벌리기)운동, 삼킴(연하)운동 등\n'
 '에 따라 종합적으로 판단하여 결정한다.\n'
 '2) ‘씹어먹는 기능에 심한 장해를 남긴 때’라 함은 심한 개구(입벌리기)운\n'
 '동 제한이나 저작(씹기)운동 제한으로 물이나 이에 준하는 음료 이외는\n'
 '섭취하지 못하는 경우를 말한다.\n'
 '3) ‘씹어먹는 기능에 뚜렷한 장해를 남긴 때’라 함은 아래의 경우 중 하나\n'
 '이상에 해당되는 때를 말한다.\n'
 '가) 뚜렷한 개구(입벌리기)운동 제한 또는 뚜렷한 저작(씹기)운동 제한-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000858',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
