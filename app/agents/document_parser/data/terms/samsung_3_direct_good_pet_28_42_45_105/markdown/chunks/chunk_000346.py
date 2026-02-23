from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제12조(계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다.\n'
 '- ⑥ 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 미쳤\n'
 '- 음을 회사가 증명하지 못한 경우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지\n'
 '- 급합니다.\n'
 '- ⑦ 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 이 특별약관을 해\n'
 '- 지하거나 보험금 지급을 거절하지 않습니다.\n'
 '- ⑧ 제22조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에 따라 이 특별'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
