from langchain_core.documents import Document

chunk = Document(
    page_content=('| 피보험자 | 보험사고의 대상이 되는 사람을 말합니다. |\n'
 '54 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)2.| 지급사유 및 보상 | 관련 용어 |\n'
 '| --- | --- |\n'
 '| 용 어 상해 | 정 의 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 신 체(의수, 의족, 의안, 의치 등 신체보조장구는 '
 '제외하나, 인공장기나 부분 의치 등 신체에 이식되어 그 기능을 대신 할 경우는 포함합니다)에 입은 상해를 말합니다. |\n'
 '| 장해 | 【별표1】(장해분류표)에서 정한 기준에 따른 장해상태를 말합니다. |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
