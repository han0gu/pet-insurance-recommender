from langchain_core.documents import Document

chunk = Document(
    page_content=('- 치료비에서 차감합니다.\n'
 '- ② 보험기간 중에 발생한 사고로 회사가 지급하는 치료비보험금의 총 합계는 보험증권에 기재된 총보\n'
 '- 상한도액을 한도로 합니다.\n'
 '【치료비보험금】 아래 ①과 ② 중 적은 금액- (피보험자가 부담한 1일당 치료비 - 1일당 자기부담금) × 보상비율\n'
 '- ② 보험증권에서 정한 1일당 보상한도액\n'
 '# 제10조(보험금의 분담)① 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경\n'
 '우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
