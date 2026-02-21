from langchain_core.documents import Document

chunk = Document(
    page_content=('수익자</td><td>보험금 지급사유가 발생하는 때에 회사에 보 험금을 청구하여 받을 수 있는 사람을 말합니 '
 '다.</td></tr><tr><td>보험증권</td><td>계약의 성립과 그 내용을 증명하기 위하여 회 사가 계약자에게 드리는 증서를 '
 '말합니다.</td></tr><tr><td>진단계약</td><td>계약을 체결하기 위하여 피보험자가 건강진단 을 받아야 하는 계약을 '
 '말합니다.</td></tr><tr><td>피보험자</td><td>보험사고의 대상이 되는 사람을'),
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
