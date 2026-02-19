from langchain_core.documents import Document

chunk = Document(
    page_content=('Chart Type: bar\n'
 '보상 | 보상제외\n'
 '틔원없이 계속 입원 | 0.06년 | 0.02년\n'
 '치고된 최종 입원일 | 0.08년 | 0.05년\n'
 '려견 위탁비용을 계속 보장합니다.\n'
 '⑧ 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는 반려견 위탁비용의 전부 또는 일부를 지급하지 '
 '않습니다.\n'
 '제2조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000602',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
