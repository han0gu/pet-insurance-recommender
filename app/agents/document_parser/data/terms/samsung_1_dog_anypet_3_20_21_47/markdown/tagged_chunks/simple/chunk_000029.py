from langchain_core.documents import Document

chunk = Document(
    page_content=('| 지급기일의 91일 이후 기간 | 보험계약대출이율+가산이율(8.0%) |\n'
 '주) 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.- 8 -당신에게 좋은보험 삼성화재# 제9조(보험금 등의 '
 '지급한도)- ① 제4조(보상하는 손해)에서 정한 치료비보험금은 보험증권에 기재된 자기부담금을 차감 후 보상비율\n'
 '- 을 곱한 금액이며 보험증권에 정한 1일당 보상한도액을 한도로 보상합니다. 자기부담금은 1일당\n'
 '- 치료비에서 차감합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
