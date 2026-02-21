from langchain_core.documents import Document

chunk = Document(
    page_content=("N96~N98에 해당하는 질병을<br>말합니다.</p><br><h1 id='45' style='font-size:16px'>⑤ 전쟁, "
 "외국의 무력행사, 혁명, 내란, 사변, 폭동</h1><br><p id='46' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 다른 약정이 없으면 피보험자가 직업, 직무 또는<br>동호회 활동목적으로 "
 "아래에 열거된 행위로 인하여 제3조</p><footer id='47' style='font-size:14px'>51</footer><p "
 "id='48'"),
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
