from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타 신체부위를 복합적으로 침범한 사항(T00.8,</td><td>T01.8, T02.8, '
 "T03.8,</td></tr></tbody></table><br><p id='30' data-category='paragraph' "
 "style='font-size:16px'>T04.8)은 얼굴, 머리, 목부위와 다른 부위의 상해가 동일사고로 인하여 중복<br>발생되는 "
 '경우에 한하여 보상됩니다.<br>2'),
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
