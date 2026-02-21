from langchain_core.documents import Document

chunk = Document(
    page_content=('심리학적 평가보고서는 정신건강의학과 의료기관에<br>서 실시되어져야 하며, 자격을 갖춘 임상심리전문<br>가가 시행하고 작성하여야 '
 '한다.<br>차) 정신행동장해 진단 전문의는 정신건강의학과 전문<br>의를 말한다.<br>카) 정신행동장해는 뇌의 기능 및 결손을 입증할 '
 '수 있<br>는 뇌자기공명촬영, 뇌전산화촬영, 뇌파 등 객관적<br>근거를 기초로 평가한다'),
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
