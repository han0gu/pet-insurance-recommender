from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조(계약 전 알릴 의무)계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건강진단할 때를 말합니다)\n'
 '청약서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 "\n'
 '계약 전 알릴의무"라 하며, 상법상 "고지의무"와 같습니다) 합니다. 반관 련 법 규 상법∙ 상법 제651조(고지의무위반으로 인한 '
 '계약해지)보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하 여\n'
 '중요한 사항을 고지하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실'),
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
