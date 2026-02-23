from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 아래와 같이</p><br><p id='33' data-category='paragraph' "
 "style='font-size:16px'>- 94 -</p><table id='34' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>질병입원일당이 중인 "
 '경우에는</td><td>지급된 최종입원일부터 180일이 경과하도록 퇴원없이 계속 입원 입원일당이 지급된 최종입원일의 그 '
 '다음날을</td></tr><tr><td></td><td>퇴원일로 봅니다. <figure><img alt="예 시'),
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
