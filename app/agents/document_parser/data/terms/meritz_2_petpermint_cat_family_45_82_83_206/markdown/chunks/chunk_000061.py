from langchain_core.documents import Document

chunk = Document(
    page_content=('(효력회복))에 따라 이 계약이 부활이 이루어진 경우에는\n'
 '부활계약을 제2항의 최초계약으로 봅니다.(부활(효력회복)\n'
 '이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을\n'
 '최초계약으로 봅니다)# 제18조(사기에 의한 계약)\uf000 계약자 또는 피보험자가 대리진단, 약물사용을 수단으로\n'
 '진단절차를 통과하거나 진단서 위·변조 또는 청약일 이전\n'
 '에 암 또는 인간면역결핍바이러스(HIV) 감염의 진단 확정을\n'
 '받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성\n'
 '립되었음을 회사가 증명하는 경우에는 계약일부터 5년 이내'),
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
