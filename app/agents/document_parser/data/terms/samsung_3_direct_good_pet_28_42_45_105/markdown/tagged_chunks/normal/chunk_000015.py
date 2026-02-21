from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 이전의 후유장해가 보험금 지급사유에 해당되지 않은 경우라도,\n'
 '- 보험금이 지급되었다면 이전의 후유장해에 해당하는 보험금\n'
 '※ 보험금 지급사유에 해당되지 않은 경우란 장해의 원인이 보장개시 이전에 발생했거나 약관상\n'
 '보험금을 지급하지 않는 사유에 해당하는 경우 등을 말합니다.# 제5조 (보험금을 지급하지 않는 사유)① 회사는 다음 중 어느 한 가지로 '
 '보험금 지급사유가 발생한 때에는 보험금을 지급하지\n'
 '않습니다.- 1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 자유로운'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
