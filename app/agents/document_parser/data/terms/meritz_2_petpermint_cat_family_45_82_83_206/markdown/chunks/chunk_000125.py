from langchain_core.documents import Document

chunk = Document(
    page_content=('기본계약 해약환급금과 기본계약 적립부분 해약환급금 중\n'
 '적은 금액의 80% 범위 내에서 신청할 수 있습니다. 중도인\n'
 '출금의 총 누적액(중도인출 원금과 이자의 합계액을 말합니\n'
 '다)은 중도인출금을 한번도 지급하지 않았을 경우의 기본계\n'
 '약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 금\n'
 '액의 80%를 한도로 합니다. 다만, 이 계약에서 정한 보험계\n'
 '약대출금이 있는 때에는 그 원금과 이자의 합계액을 공제한\n'
 '후의 잔액을 기준으로 합니다.\uf000 제1항의 중도인출금을 지급받은 경우에는「보험료 및 해'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
