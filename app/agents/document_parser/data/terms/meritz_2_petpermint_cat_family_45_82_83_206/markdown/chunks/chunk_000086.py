from langchain_core.documents import Document

chunk = Document(
    page_content=('의 뒷면에 기재하여 드립니다.- ① 보험종목\n'
 '- ② 보험기간\n'
 '- ③ 보험료 납입주기, 납입방법 및 납입기간\n'
 '- ④ 계약자, 피보험자\n'
 '- ⑤ 보험가입금액, 보험료 등 기타 계약의 내용\n'
 '\uf000 계약자는 보험수익자를 변경할 수 있으며 이 경우에는\n'
 '회사의 승낙이 필요하지 않습니다. 다만, 변경된 보험수익\n'
 '자가 회사에 권리를 대항하기 위해서는 계약자가 보험수익\n'
 '자가 변경되었음을 회사에 통지하여야 합니다.68# 【부가설명】계약자가 보험수익자가 변경되었음을 회사에 통지하기 전\n'
 '에 보험금 지급사유가 발생한 경우 회사는 변경 전 보험'),
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
