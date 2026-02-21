from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 뼈에 가관절이 남은 경우를 말한다.\n'
 '- 13) ‘뼈에 기형을 남긴 때’라 함은 상완골 또는 요골과 척골에 변형이 남아\n'
 '- 정상에 비해 부정유합된 각 변형이 15° 이상인 경우를 말한다.\n'
 '148 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- \n'
 '# 다.- 지급률의 결정\n'
 '- 1) 한 팔의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에\n'
 '- 기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다.\n'
 '- 2) 1상지(팔과 손가락)의 후유장해지급률은 원칙적으로 각각 합산하되, 지\n'
 '# 급률은 60% 한도로 한다.-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000899',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
