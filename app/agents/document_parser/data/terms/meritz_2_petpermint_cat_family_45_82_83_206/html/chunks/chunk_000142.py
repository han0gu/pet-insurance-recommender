from langchain_core.documents import Document

chunk = Document(
    page_content=(". 서명자가 해당 전자문서에 서명하였다는 사실</p><br><p id='99' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제3항에도 불구하고 전화를 이용하여 계약을 체결하는<br>경우 다음의 각 호의 어느 "
 '하나를 충족하는 때에는 자필서<br>명을 생략할 수 있으며, 제2항의 규정에 따른 음성녹음 내<br>용을 문서화한 확인서를 계약자에게 '
 "드림으로써 계약자 보<br>관용 청약서를 전달한 것으로 봅니다.</p><br><p id='100' data-category='list'"),
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
