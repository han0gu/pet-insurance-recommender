from langchain_core.documents import Document

chunk = Document(
    page_content=('다리 등)는 해당 장해로도 평가하고 그 중 높은 지급률을 적용한다.\n'
 '라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발병 또는 외상 후 12\n'
 '개월 동안 지속적으로 치료한 후에 장해를 평가한다. 그러나, 12개\n'
 '월이 지났다고 하더라도 뚜렷하게 기능 향상이 진행되고 있는 경우- \n'
 '표별KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 153- 153 -- \n'
 '# 또는 단기간내에 사망이 예상되는 경우는 6개월의 범위에서 장해 평가를 유보한다.마) 장해진단 전문의는 재활의학과, 신경외과 또는 '
 '신경과 전문의로 한다.- 2) 정신행동'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000933',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
