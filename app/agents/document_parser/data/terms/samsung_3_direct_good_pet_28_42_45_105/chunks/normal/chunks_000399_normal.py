from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 제1항 제2호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에는 제 제12조(계약 후 알릴 의무) 제4항 또는 '
 '제5항에 따라 보험금을 지급합니다. ⑥ 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 미쳤 음을 회사가 '
 '증명하지 못한 경우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지 급합니다. ⑦ 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 '
 '위반을 이유로 이 특별약관을 해 지하거나 보험금 지급을 거절하지 않습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000399',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
