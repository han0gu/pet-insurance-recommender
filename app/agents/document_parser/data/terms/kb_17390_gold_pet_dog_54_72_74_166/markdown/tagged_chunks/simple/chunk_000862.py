from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 가운데 앞니(중절치)간 거리를 기준으로 한다. 단, 가운데 앞니(중절\n'
 '- 치)가 없는 경우에는 측정가능한 인접 치아간 거리의 최대치를 기준으로\n'
 '- 한다.\n'
 '- 6) 부정교합은 위턱(상악)과 아래턱(하악)의 부조화로 윗니(상악치아)와 아 특별\n'
 '- 랫니(하악치아)가 전방 및 측방으로 맞물림에 제한이 있는 상태를 말한다. 약\n'
 '- 7) ‘말하는 기능에 심한 장해를 남긴 때’라 함은 아래의 경우 중 하나 이 관\n'
 '- 상에 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 30%미만인 경우'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000862',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
