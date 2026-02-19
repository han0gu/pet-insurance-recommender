from langchain_core.documents import Document

chunk = Document(
    page_content=('다른 계약이 없을 때 이 계약의 보상책임액 손해액(피보험자가 부담한 총비용) × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 '
 '합계액\n'
 '【예시】 두 보험회사와 계약을 체결하고 100만원의 손해가 발생한 경우 보험금 계산 예시는 아래와 같습 니다. A사만 가입한 경우 A사의 '
 '보상책임액이 90만원이고 B사만 가입한 경우 B사의 보상책임액이 60만원 인 경우, A사 : 100만원 × 90만원 / (90만원 + '
 '60만원) = 60만원 지급 B사 : 100만원 × 60만원 / (90만원 + 60만원) = 40만원 지급'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000033',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
