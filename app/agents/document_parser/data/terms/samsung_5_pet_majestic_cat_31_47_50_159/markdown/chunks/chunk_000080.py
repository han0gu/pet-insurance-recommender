from langchain_core.documents import Document

chunk = Document(
    page_content=('중 먼저 도래하는 기간을 말합니다.- ③ 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우편,\n'
 '- 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시(이하 ‘서면 등’이라 합니\n'
 '- 다)를 발송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한 때에 그 발송 사실을\n'
 '- 회사에 지체없이 알려야 합니다.\n'
 '- ④ 계약자가 청약을 철회한 때에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내에\n'
 '- 납입한 보험료를 계약자에게 돌려 드리며, 보험료 반환이 늦어진 기간에 대하여는 이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
