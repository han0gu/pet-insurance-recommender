from langchain_core.documents import Document

chunk = Document(
    page_content=('배려하고 신뢰에 어긋나지 않도록 행동해야 한다는 원칙을 말합니다.【관련법규】- < 「민법」 제2조(신의성실))>\n'
 '- ① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.\n'
 '- ② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다.\n'
 '- ③ 회사는 보상하지 않는 손해 등 계약자나 피보험자에게 불리하거나 부담을 주는 내용은 확대하여\n'
 '- 해석하지 않습니다.\n'
 '제35조(설명서 교부 및 보험안내자료 등의 효력)- ① 회사는 일반금융소비자에게 청약을 권유하거나 일반금융소비자가 설명을 요청하는 경우 '
 '보험상품'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
