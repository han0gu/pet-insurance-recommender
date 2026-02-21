from langchain_core.documents import Document

chunk = Document(
    page_content=('발생하기 전에 피보험자가 서면(「전자서명법」 제2<br>조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제<br>44조의2에 '
 '정하는 바에 따라 본인 확인 및 위조ㆍ변조 방지<br>에 대한 신뢰성을 갖춘 전자문서를 포함)으로 '
 '동의하여야<br>합니다.<br>\uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계<br>약자에게 보험증권 및 약관을 '
 "교부하고 변경된 계약자가 요<br>청하는 경우 약관의 중요한 내용을 설명하여 드립니다.</p><h1 id='23' "
 "style='font-size:20px'>제24조(보험나이"),
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
