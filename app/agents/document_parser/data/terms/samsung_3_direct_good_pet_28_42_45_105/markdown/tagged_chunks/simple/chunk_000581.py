from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사」라 합니다)가 정한 기준에 적합하지 않은 경우 보험계약자(이하「계약자」라 합니\n'
 '- 다)의 청약과 회사의 승낙으로 보험계약에 부가하여 이루어 집니다.\n'
 '- ② 이 특별약관에 대한 보장개시일(책임개시일)은 보험계약「제1회 보험료 및 회사의 보\n'
 '- 장개시」의 보장개시일(책임개시일)과 동일합니다.\n'
 '- ③ 보험계약이 해지 또는 기타 사유에 의하여 효력이 없게 된 경우에는 이 특별약관도 더\n'
 '- 이상 효력이 없습니다.\n'
 '# 제 2조 (특별약관의 내용)이 특별약관은 피보험자의 위험도가 높아 계약이 불가능한 경우 이 특별약관이 정하는'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
