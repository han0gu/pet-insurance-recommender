from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대\n'
 '- 한 것인\xa0경우에는 그 권리를 취득하지 못합니다. 다만,\xa0손해가\xa0그 가족의 고의로\n'
 '# 인하여\xa0발생한 경우에는 그 권리를 취득합니다.제16조(계약 전 알릴 의무)\n'
 '계약자, 피보험자 또는 이들의 대리인은 청약할 때 청약서(질문서를 포함합니다)에\n'
 '서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 "계약 전\n'
 '알릴 의무"라 하며, 상법상 "고지의무"와 같습니다)합니다.| 관 련 법 | 규 상법 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
