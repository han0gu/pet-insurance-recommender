from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 청구하여야 하\n'
 '- 며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해약환급\n'
 '- 금 지급일까지의 기간에 대한 이자의 계산은 보험금을 지급할 때의 적립이율 계산([별\n'
 '- 표1] 참조)에 따릅니다.\n'
 '- ③ 제21 조(계약내용의 변경 등) 제1항 제5호에서 정한 적립보험료 등을 감액할 경우 제1\n'
 '- 항에 정한 해약환급금은 없거나 최초가입시 안내한 금액보다 적어질 수 있습니다.'),
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
