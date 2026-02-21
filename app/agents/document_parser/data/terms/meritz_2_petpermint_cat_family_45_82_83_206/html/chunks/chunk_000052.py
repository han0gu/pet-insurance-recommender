from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 정<br>당한 사유 없이 이에 동의하지 않을 경우 사실확인이 끝날<br>때까지 회사는 보험금지급지연에 따른 이자를 지급하지 '
 '않<br>습니다.<br>\uf000 회사는 제6항의 서면조사에 대한 동의 요청시 조사목적,<br>사용처 등을 명시하고 '
 "설명합니다.</p><h1 id='70' style='font-size:20px'>제9조(적립부분 적립이율에 관한 "
 "사항)</h1><br><p id='71' data-category='paragraph' "
 "style='font-size:16px'>\uf000 이 보험의 적립부분 순보험료(적립보험료에서"),
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
