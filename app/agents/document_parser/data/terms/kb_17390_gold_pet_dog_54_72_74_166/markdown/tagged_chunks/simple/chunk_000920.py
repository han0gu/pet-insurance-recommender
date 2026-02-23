from langchain_core.documents import Document

chunk = Document(
    page_content=('- 까운 쪽을, 나머지 네 발가락에서는 제1지관절(근위지관절)부터(제1지\n'
 '- 관절 포함) 심장에서 가까운 쪽을 잃었을 때를 말한다.\n'
 '- 4) 리스프랑 관절 이상에서 잃은 때라 함은 족근-중족골간 관절 이상에서\n'
 '- 절단된 경우를 말한다.\n'
 '- 5) ‘발가락뼈 일부를 잃었을 때’라 함은 첫째 발가락의 지관절, 다른 네\n'
 '- 발가락의 제1지관절(근위지관절)부터 심장에서 먼 쪽으로 발가락 뼈 일\n'
 '- 부가 절단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 발가락 길이\n'
 '- 의 단축 없이 골편만 떨어진 상태는 해당하지 않는다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000920',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
